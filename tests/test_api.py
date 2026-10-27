"""/api/predict, /api/odds, /api/teams/{id}/form and /availability with stubbed sources."""
import json
from datetime import timedelta

import numpy as np
import pytest

from config import FEATURE_COLS
from data.schedule import parse_scoreboard
from server import clock, model_store
from server.services import demo, espn, market, predict

LAL, GSW, LAC, BOS = 1610612747, 1610612744, 1610612746, 1610612738
SECRET = "sk-test-SECRET-0123456789"


def team_row(name, team_id, strength=0.0):
    """A get_team_advanced_stats-shaped dict; strength shifts the form up or down."""
    s = strength
    return {
        "name": name, "team_id": team_id, "ELO": 1500.0,
        "NET5": 2.0 + s, "NET15": 1.5 + s, "PTS10": 114.0 + s, "PA10": 111.0 - s,
        "EFG10": 0.54 + s / 200, "TOV10": 0.13, "ORB10": 0.25, "FTR10": 0.21,
        "PACE10": 99.0, "WPCT10": 0.6, "WIN_STREAK": 2, "DAYS_REST": 2, "IS_B2B": 0,
        "HOME_WPCT": 0.65, "AWAY_WPCT": 0.5, "PLUS_MINUS": 3.0 + s, "REB": 44.0,
        "eFG_PCT": 0.54, "AST_TOV": 1.8, "TS_PCT": 0.58, "OREB_PCT": 0.25,
        "PTS": 114.0 + s, "PTS_STD": 11.0, "PTS_ALLOWED": 111.0 - s, "PTS_ALLOWED_STD": 10.0,
    }


def espn_board(state="pre", odds=True):
    comp = {
        "competitors": [
            {"homeAway": "home", "score": "0", "team": {"displayName": "Los Angeles Lakers", "abbreviation": "LAL"}},
            {"homeAway": "away", "score": "0", "team": {"displayName": "Golden State Warriors", "abbreviation": "GS"}},
        ],
        "odds": [{"provider": {"name": "DraftKings"}, "details": "LAL -4.5", "overUnder": 226.5,
                  "spread": -4.5, "homeTeamOdds": {"moneyLine": -180}, "awayTeamOdds": {"moneyLine": 150}}]
        if odds else [],
    }
    if state == "post":
        comp["competitors"][0]["score"], comp["competitors"][1]["score"] = "118", "109"
    return {"events": [{"id": "401700001", "date": "2026-10-28T02:30Z",
                        "status": {"period": 4 if state == "post" else 0,
                                   "type": {"state": state, "completed": state == "post"}},
                        "competitions": [comp]}]}


INJURY_FEED = {"injuries": [
    {"displayName": "Los Angeles Lakers", "injuries": [
        {"athlete": {"displayName": "LeBron James"}, "status": "Out", "shortComment": "Ankle"},
        {"athlete": {"displayName": "Some Rookie"}, "status": "Questionable", "shortComment": "Knee"},
        {"athlete": {"displayName": "Healthy Guy"}, "status": "Active"}]},
    {"displayName": "LA Clippers", "injuries": [
        {"athlete": {"displayName": "Kawhi Leonard"}, "status": "Day-To-Day", "shortComment": ""}]},
]}


class StubModel:
    """predict_proba like a calibrated classifier: higher Elo/net diff → higher P(home)."""

    def __init__(self):
        self.rows = []

    def predict_proba(self, row):
        self.rows.append(row.copy())
        z = row["ELO_DIFF"].iloc[0] / 200 + row["NET15_DIFF"].iloc[0] / 10
        p = 1 / (1 + np.exp(-z))
        return np.array([[1 - p, p]])


@pytest.fixture
def sources(monkeypatch):
    """Every external source stubbed; no network, no nba_cache.db."""
    mem = demo.MemoryCache()
    calls = {"scoreboard": 0, "injuries": 0, "odds_api": 0}
    state = {"board": espn_board(), "feed": INJURY_FEED, "model": None,
             "stats": {"Los Angeles Lakers": 2.0, "Golden State Warriors": -1.0,
                       "Los Angeles Clippers": 0.0, "Boston Celtics": 1.0}}

    def fake_stats(name, target_date=None, window=10):
        from server.services import teams
        team = teams.resolve(name)
        return team_row(team["name"], team["id"], state["stats"][team["name"]])

    def fake_board(d):
        calls["scoreboard"] += 1
        if isinstance(state["board"], Exception):
            raise state["board"]
        return state["board"]

    def fake_feed():
        calls["injuries"] += 1
        if isinstance(state["feed"], Exception):
            raise state["feed"]
        return state["feed"]

    def fake_odds_api(key):
        calls["odds_api"] += 1
        raise RuntimeError(f"403 Client Error for url: https://api.the-odds-api.com/v4/?apiKey={key}")

    from engine import features
    monkeypatch.setattr(features, "get_team_advanced_stats", fake_stats)
    monkeypatch.setattr(espn, "_fetch_scoreboard", fake_board)
    monkeypatch.setattr(espn, "_fetch_injury_feed", fake_feed)
    monkeypatch.setattr(espn, "_cache", mem)
    monkeypatch.setattr(market, "_cache", mem)
    monkeypatch.setattr(market, "_fetch_odds_api_all", fake_odds_api)
    monkeypatch.setattr(predict, "_cache", mem)
    monkeypatch.setattr(predict, "_load_elo", lambda: {
        "Los Angeles Lakers": 1600.0, "Golden State Warriors": 1450.0,
        "LA Clippers": 1530.0, "Boston Celtics": 1650.0})
    monkeypatch.setattr(model_store, "get_model", lambda: state["model"])
    state["calls"], state["cache"] = calls, mem
    return state


def strict_json(text):
    def bad(c):
        raise AssertionError(f"non-finite number {c} in JSON")
    return json.loads(text, parse_constant=bad)


def post_predict(client, **body):
    body.setdefault("home_id", LAL)
    body.setdefault("away_id", GSW)
    return client.post("/api/predict", json=body)


TOP_KEYS = {"as_of", "mode", "model", "game", "win_prob", "pick", "flags", "engines", "ensemble",
            "margin", "total", "scores", "market", "fair_line", "kelly", "teams", "ratings",
            "availability", "radar", "feature_row", "caveats"}


# ── predict: elo_only ────────────────────────────────────────────────────────

def test_predict_elo_only_schema(client, sources):
    r = post_predict(client, include=["hist", "poisson", "shap"])
    assert r.status_code == 200, r.text
    d = strict_json(r.text)
    assert TOP_KEYS <= set(d) and {"poisson", "shap"} <= set(d)
    assert d["mode"] == "elo_only" and d["model"] is None and d["shap"] is None
    assert "elo_only_no_model" in d["caveats"]
    assert d["engines"]["xgb"] is None
    h_elo, a_elo = d["teams"]["home"]["adjusted"]["elo"], d["teams"]["away"]["adjusted"]["elo"]
    assert d["engines"]["elo"] == pytest.approx(1 / (1 + 10 ** ((a_elo - h_elo - 70) / 400)), abs=1e-4)
    assert d["win_prob"]["home"] == d["engines"]["elo"]
    assert d["win_prob"]["home"] + d["win_prob"]["away"] == pytest.approx(1, abs=1e-4)
    ens = d["ensemble"]
    assert [c["key"] for c in ens["components"]] == ["elo"] and ens["components"][0]["coef"] == 1.0
    assert ens["final"]["prob"] == d["win_prob"]["home"]
    assert d["pick"] == {"side": "home", "team_id": LAL, "prob": d["win_prob"]["home"], "tier": "high"}
    assert d["game"]["home"]["abbr"] == "LAL" and d["game"]["away"]["abbr"] == "GSW"
    assert d["game"]["event_id"] == "401700001" and d["game"]["tipoff_utc"] == "2026-10-28T02:30:00Z"
    assert d["ratings"]["home"]["elo"] == 1600.0 and d["ratings"]["home"]["elo_rank"] == 2
    assert d["ratings"]["away"]["kalman"] is None
    assert set(d["feature_row"]) == set(FEATURE_COLS)
    assert d["feature_row"]["ELO_DIFF"] < 150                 # injuries applied (LeBron out)
    assert len(d["radar"]["axes"]) == 6 and len(d["radar"]["home"]) == 6
    assert len(d["poisson"]["quarters"]) == 4 and d["engines"]["poisson"] is not None


def test_predict_distributions(client, sources):
    d = post_predict(client).json()
    m, t = d["margin"], d["total"]
    assert m["home_spread"] == pytest.approx(-m["mean"], abs=0.11)
    assert len(m["hist"]["edges"]) == 61 and len(m["hist"]["counts"]) == 60
    assert sum(m["hist"]["counts"]) == 50_000
    lo50, hi50 = m["intervals"]["p50"]
    lo80, hi80 = m["intervals"]["p80"]
    lo95, hi95 = m["intervals"]["p95"]
    assert lo95 < lo80 < lo50 < m["mean"] < hi50 < hi80 < hi95
    assert 0 < m["p_within_3"] < 0.5
    assert t["intervals"]["p50"][0] < t["median"] < t["intervals"]["p50"][1]
    assert len(t["hist"]["edges"]) == 61
    assert d["scores"]["home"]["p25"] < d["scores"]["home"]["mean"] < d["scores"]["home"]["p75"]
    assert "hist" in d["scores"]["away"]
    # include=[] drops the histograms
    d = post_predict(client, include=[]).json()
    assert "hist" not in d["margin"] and "poisson" not in d and "shap" not in d


def test_predict_market_from_espn(client, sources):
    d = post_predict(client).json()
    mk = d["market"]
    assert mk["status"] == "ok" and mk["source"] == "DraftKings via ESPN"
    assert (mk["home_ml"], mk["away_ml"], mk["spread"], mk["total"]) == (-180, 150, -4.5, 226.5)
    assert mk["fair"]["home"] + mk["fair"]["away"] == pytest.approx(1, abs=1e-3)
    assert mk["edge"]["home"] == pytest.approx(d["win_prob"]["home"] - mk["fair"]["home"], abs=1e-3)
    assert 0 < mk["cover_prob"] < 1 and 0 < mk["over_prob"] < 1
    assert d["kelly"]["price_source"] == "market"
    assert d["kelly"]["home"]["decimal"] == pytest.approx(1 + 100 / 180, abs=1e-4)
    p = d["win_prob"]["home"]
    assert d["fair_line"]["home_ml"] == pytest.approx(-100 * p / (1 - p), abs=0.2)


def test_predict_manual_odds(client, sources):
    d = post_predict(client, odds={"home_ml": 120, "away_ml": -140}).json()
    assert d["market"]["status"] == "manual" and d["market"]["home_ml"] == 120
    assert d["market"]["spread"] == -4.5                     # still from the market lookup
    assert d["kelly"]["price_source"] == "manual"
    assert d["kelly"]["home"]["frac_pct"] <= d["kelly"]["cap_pct"]


def test_predict_without_market(client, sources):
    sources["board"] = espn_board(odds=False)
    d = post_predict(client).json()
    assert d["market"]["status"] == "no_odds_yet" and d["kelly"] is None
    assert d["flags"]["value"] is False


def test_predict_offline_scoreboard_without_cache(client, sources):
    sources["board"] = ConnectionError("offline")
    d = post_predict(client).json()
    assert d["market"] is None and d["kelly"] is None and d["game"]["event_id"] is None


def test_predict_past_final_and_caveat(client, sources):
    sources["board"] = espn_board(state="post")
    past = (clock.today_et() - timedelta(days=3)).isoformat()
    d = post_predict(client, date=past).json()
    assert d["game"]["date"] == past and d["game"]["result"] == {"home_score": 118, "away_score": 109}
    assert "past_date_current_elo" in d["caveats"]
    assert "injuries_unavailable" in d["caveats"]               # no report for past dates
    assert d["availability"]["home"]["players"] == []


# ── predict: blend_v5 ────────────────────────────────────────────────────────

def test_predict_blend_ensemble_sums_to_final(client, sources):
    from engine.models import get_calibrator
    stub = StubModel()
    sources["model"] = stub
    d = post_predict(client, injuries="none").json()
    assert d["mode"] == "blend_v5" and "elo_only_no_model" not in d["caveats"]
    ens = d["ensemble"]
    assert ens["order"] == ["elo", "mc", "xgb"] == [c["key"] for c in ens["components"]]
    total = ens["base"]["logit"] + sum(c["contribution_logit"] for c in ens["components"])
    assert total == pytest.approx(ens["final"]["logit"], abs=1e-3)
    steps = sum(c["step_pp"] for c in ens["components"]) / 100
    assert steps == pytest.approx(ens["final"]["prob"] - ens["base"]["prob"], abs=1e-3)
    assert sum(c["share"] for c in ens["components"]) == pytest.approx(1, abs=1e-3)
    assert ens["final"]["prob"] == d["win_prob"]["home"]

    calib = get_calibrator()
    x = {"xgb": d["engines"]["xgb"], "mc": d["engines"]["mc"],
         "elo": d["feature_row"]["ELO_DIFF"] / 400}
    expected = calib.predict(x["xgb"], x["mc"], x["elo"] * 400)
    assert d["win_prob"]["home"] == pytest.approx(expected, abs=2e-4)
    w = dict(zip(("xgb", "mc", "elo"), calib.weights))
    for c in ens["components"]:
        assert c["coef"] == pytest.approx(w[c["key"]], abs=1e-4)
    assert ens["base"]["logit"] == pytest.approx(0.5 * w["xgb"] + 0.5 * w["mc"] + calib.bias, abs=1e-4)
    assert d["engines"]["xgb"] == pytest.approx(stub.predict_proba(stub.rows[0])[0][1], abs=1e-4)


def test_blend_ladder_math():
    w, b = {"xgb": 2.0, "mc": 1.5, "elo": 0.3}, -1.75
    lad = predict.blend_ladder(w, b, {"xgb": 0.7, "mc": 0.6, "elo": 0.5}, {"xgb": 0.7, "mc": 0.6, "elo": 0.66})
    assert lad["base"]["logit"] == pytest.approx(0.0) and lad["base"]["prob"] == 0.5
    c = {x["key"]: x for x in lad["components"]}
    assert c["xgb"]["contribution_logit"] == pytest.approx(0.4)
    assert c["mc"]["contribution_logit"] == pytest.approx(0.15)
    assert c["elo"]["contribution_logit"] == pytest.approx(0.15)
    assert lad["final"]["logit"] == pytest.approx(0.7)
    assert c["xgb"]["share"] == pytest.approx(0.4 / 0.7, abs=1e-4)
    assert lad["components"][0]["step_pp"] == pytest.approx((1 / (1 + np.exp(-0.15)) - 0.5) * 100, abs=0.01)


def test_predict_conflict_suppresses_pick(client, sources):
    class Contrarian(StubModel):
        def predict_proba(self, row):
            return np.array([[0.8, 0.2]])                   # away, while MC and Elo say home
    sources["model"] = Contrarian()
    sources["stats"]["Los Angeles Lakers"] = 8.0
    d = post_predict(client, injuries="none").json()
    assert d["engines"]["mc"] > 0.5 and d["engines"]["xgb"] < 0.5
    assert d["flags"]["conflict"] is True
    assert not d["flags"]["show_pick"] and not d["flags"]["safe"] and not d["flags"]["vip"]


def test_predict_shap_uses_adjusted_rows(client, sources):
    pytest.importorskip("shap")
    from xgboost import XGBClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURE_COLS)))
    X[:, 0] *= 100
    import pandas as pd
    X = pd.DataFrame(X, columns=FEATURE_COLS)
    y = (X["ELO_DIFF"] + rng.normal(scale=50, size=200) > 0).astype(int)
    model = XGBClassifier(n_estimators=8, max_depth=2, n_jobs=1).fit(X, y)
    sources["model"] = model
    d = post_predict(client, include=["shap"]).json()
    s = d["shap"]
    assert s is not None and len(s["features"]) == len(FEATURE_COLS)
    by_name = {f["name"]: f for f in s["features"]}
    # SHAP explains exactly the injury-adjusted row the model saw (E7)
    for name, value in d["feature_row"].items():
        assert by_name[name]["value"] == pytest.approx(value, abs=1e-4)
    assert abs(s["features"][0]["shap"]) >= abs(s["features"][-1]["shap"])


# ── injuries ─────────────────────────────────────────────────────────────────

def test_live_injuries_lower_strength(client, sources):
    d = post_predict(client).json()
    av = d["availability"]["home"]
    assert av["status"] == "ok" and av["source"] == "live" and av["units"] == "factor"
    names = [p["name"] for p in av["players"]]
    assert names == ["LeBron James", "Some Rookie"]              # sorted by impact × weight
    lebron = av["players"][0]
    assert lebron["matched"] is True and lebron["status"] == "out" and lebron["status_weight"] == 1.0
    assert av["players"][1]["matched"] is False and av["unmatched"] == 1
    assert av["current"] < av["healthy"] == 1.0 and av["deficit"] == pytest.approx(av["current"] - 1, abs=1e-4)
    assert d["teams"]["home"]["adjusted"]["elo"] < d["teams"]["home"]["elo"]
    assert d["availability"]["away"]["players"] == []
    assert sources["calls"]["injuries"] == 1                     # one feed call for both teams


def test_injuries_none_and_custom(client, sources):
    base = post_predict(client, injuries="none").json()
    assert base["availability"]["home"]["source"] == "none"
    assert base["availability"]["home"]["current"] == 1.0
    custom = post_predict(client, injuries={"home": [], "away": [
        {"name": "Stephen Curry", "status": "out"}]}).json()
    assert custom["availability"]["away"]["source"] == "custom"
    assert custom["availability"]["away"]["players"][0]["matched"] is True
    assert custom["teams"]["away"]["adjusted"]["elo"] < custom["teams"]["away"]["elo"]
    assert custom["engines"]["elo"] > base["engines"]["elo"]


def test_injury_feed_down(client, sources):
    sources["feed"] = ConnectionError("down")
    d = post_predict(client).json()
    assert d["availability"]["home"]["status"] == "unavailable"
    assert "injuries_unavailable" in d["caveats"]


def test_injury_feed_stale_copy(client, sources):
    import time
    sources["feed"] = ConnectionError("down")
    sources["cache"].set("web:espn_injuries", INJURY_FEED, ttl=-1, stored_at=time.time() - 3600)
    d = post_predict(client).json()
    assert d["availability"]["home"]["status"] == "stale" and "injuries_stale" in d["caveats"]


def test_clippers_injuries_and_elo_match_by_id(client, sources):
    av = client.get(f"/api/teams/{LAC}/availability").json()
    assert [p["name"] for p in av["players"]] == ["Kawhi Leonard"]
    form = client.get(f"/api/teams/{LAC}/form").json()
    assert form["elo"] == 1530.0                           # "LA Clippers" key in the Elo file


# ── validation and errors ────────────────────────────────────────────────────

@pytest.mark.parametrize("body,code", [
    ({"home_id": LAL, "away_id": LAL}, 422),
    ({"home_id": LAL, "away_id": 123}, 404),
    ({"home_id": LAL, "away_id": GSW, "date": "28/10/2026"}, 422),
    ({"home_id": LAL, "away_id": GSW, "odds": {"home_ml": 50, "away_ml": -110}}, 422),
    ({"home_id": LAL, "away_id": GSW, "injuries": "sometimes"}, 422),
    ({"home_id": LAL, "away_id": GSW, "injuries": {"home": [{"name": "X", "status": "maybe"}]}}, 422),
    ({"home_id": LAL, "away_id": GSW, "include": ["everything"]}, 422),
])
def test_predict_validation(client, sources, body, code):
    r = client.post("/api/predict", json=body)
    assert r.status_code == code
    err = r.json()
    assert err["error"] in ("validation", "team_not_found") and err["detail"]


def test_predict_upstream_unavailable(client, sources, monkeypatch):
    from engine import features
    monkeypatch.setattr(features, "get_team_advanced_stats", lambda *a, **k: None)
    r = post_predict(client)
    assert r.status_code == 503
    assert r.json()["error"] == "upstream_unavailable" and r.json()["source"] == "nba_api"


def test_team_logs_stale_fallback(client, sources, monkeypatch):
    """stats.nba.com down but an expired log is cached: use it, flagged stale."""
    from engine import features
    real = features.get_team_advanced_stats
    mem = sources["cache"]
    mem.set(f"lgf_{LAL}", "logs", ttl=-1, stored_at=1.0)
    mem.set(f"lgf_{GSW}", "logs", ttl=-1, stored_at=1.0)

    def flaky(name, target_date=None, window=10):
        from server.services import teams
        team = teams.resolve(name)
        return real(name, target_date) if mem.get(f"lgf_{team['id']}") is not None else None
    monkeypatch.setattr(features, "get_team_advanced_stats", flaky)
    d = post_predict(client).json()
    assert d["teams"]["home"]["stale"] is True and "stats_stale" in d["caveats"]
    assert d["teams"]["home"]["as_of"] == "1970-01-01T00:00:01Z"


def test_form_endpoint(client, sources):
    r = client.get(f"/api/teams/{LAL}/form", params={"date": "2026-01-15"})
    assert r.status_code == 200
    f = r.json()
    assert f["team"]["abbr"] == "LAL" and f["date"] == "2026-01-15"
    for key in ("elo", "net5", "net15", "efg10", "days_rest", "is_b2b", "pts", "pts_allowed", "as_of"):
        assert key in f
    assert all(k == k.lower() for k in f)
    assert client.get("/api/teams/99/form").json()["error"] == "team_not_found"


# ── odds ─────────────────────────────────────────────────────────────────────

def test_odds_endpoint_espn(client, sources):
    o = client.get("/api/odds", params={"home_id": LAL, "away_id": GSW}).json()
    assert o == {"status": "ok", "source": "DraftKings via ESPN", "fetched_at": o["fetched_at"],
                 "stale": False, "home_ml": -180, "away_ml": 150, "spread": -4.5, "total": 226.5}
    # reversed home/away is a different game
    assert client.get("/api/odds", params={"home_id": GSW, "away_id": LAL}).json()["status"] == "not_found"
    client.get("/api/odds", params={"home_id": LAL, "away_id": GSW})
    assert sources["calls"]["scoreboard"] == 1                    # cached


def test_odds_stale_scoreboard(client, sources):
    sources["cache"].set(f"web:espn_scoreboard:{clock.today_et():%Y%m%d}",
                         parse_scoreboard(espn_board()), ttl=-1, stored_at=1000.0)
    sources["board"] = ConnectionError("offline")
    o = client.get("/api/odds", params={"home_id": LAL, "away_id": GSW}).json()
    assert o["status"] == "ok" and o["stale"] is True
    d = post_predict(client).json()
    assert "odds_stale" in d["caveats"]


def test_odds_unavailable(client, sources):
    sources["board"] = ConnectionError("offline")
    r = client.get("/api/odds", params={"home_id": LAL, "away_id": GSW})
    assert r.status_code == 503 and r.json()["source"] == "espn"


def test_odds_api_used_first(client, sources, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", SECRET)
    tip = f"{clock.today_et().isoformat()}T23:30:00Z"
    games = [{"home_team": "Los Angeles Lakers", "away_team": "Golden State Warriors", "commence_time": tip,
              "bookmakers": [{"title": "FanDuel", "markets": [{"key": "h2h", "outcomes": [
                  {"name": "Los Angeles Lakers", "price": -170}, {"name": "Golden State Warriors", "price": 145}]}]}]}]
    calls = []
    monkeypatch.setattr(market, "_fetch_odds_api_all", lambda key: calls.append(key) or games)
    o = client.get("/api/odds", params={"home_id": LAL, "away_id": GSW}).json()
    assert o["source"] == "FanDuel via The Odds API" and (o["home_ml"], o["away_ml"]) == (-170, 145)
    assert o["spread"] == -4.5                                     # ESPN line alongside
    client.get("/api/odds", params={"home_id": LAL, "away_id": GSW})
    assert calls == [SECRET]                                       # one call, then cached


def test_secret_never_in_responses(client, sources, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", SECRET)
    o = client.get("/api/odds", params={"home_id": LAL, "away_id": GSW})
    assert o.json()["source"] == "DraftKings via ESPN"            # fell back after the 403
    assert sources["calls"]["odds_api"] == 1
    p = post_predict(client, include=["hist", "poisson", "shap"])
    sources["board"] = ConnectionError(f"https://x/?apiKey={SECRET}")
    sources["cache"].__init__()
    e = client.get("/api/odds", params={"home_id": LAL, "away_id": GSW})
    for r in (o, p, e):
        assert SECRET not in r.text
    assert e.status_code == 503
