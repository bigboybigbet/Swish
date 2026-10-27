"""Offline tests for the Kalman-Elo team strength model."""
import numpy as np
import pandas as pd
import pytest

from engine import team_strength as ts
from tests.test_core import _fake_logs


def _logs(n_days=120):
    raw = _fake_logs(n_days=n_days)
    raw["SEASON_ID"] = "22025"
    return raw


def test_game_table_reads_season_id_and_pairs_games():
    g = ts.game_table(_logs())
    assert (g["SEASON"] == "2025-26").all()
    assert g["GAME_ID"].is_unique
    assert set(g["RESULT"]) <= {0, 1}


def test_replay_never_uses_the_game_itself():
    g = ts.game_table(_logs())
    base = ts.replay(g)["EXP_MARGIN"].values
    k = len(g) // 2
    g2 = g.copy()
    g2.loc[k, "MARGIN"] *= -1
    flipped = ts.replay(g2)["EXP_MARGIN"].values
    np.testing.assert_allclose(base[:k + 1], flipped[:k + 1])
    assert not np.allclose(base[k + 1:], flipped[k + 1:])


def test_strongest_team_rates_highest_and_wins_more():
    state = ts.build_state(_logs(), fit_seasons=["2025-26"])
    table = ts.ratings_table(state)
    assert table.iloc[0]["TEAM"] == "Team 0"
    assert table.iloc[-1]["TEAM"] == "Team 3"
    p = ts.predict(state, "Team 0", "Team 3", "2026-02-20")
    assert p["p_home"] > 0.7 and p["exp_margin"] > 0
    q = ts.predict(state, "Team 3", "Team 0", "2026-02-20")
    assert q["p_home"] < p["p_home"]


def test_injuries_lower_the_injured_side():
    state = ts.build_state(_logs(), fit_seasons=["2025-26"])
    healthy = ts.predict(state, "Team 1", "Team 2", "2026-02-20")
    hurt = ts.predict(state, "Team 1", "Team 2", "2026-02-20", home_pts_lost=5.0)
    assert hurt["exp_margin"] == pytest.approx(healthy["exp_margin"] - 5.0)
    assert hurt["p_home"] < healthy["p_home"]


def test_new_season_regresses_ratings():
    state = ts.build_state(_logs(), fit_seasons=["2025-26"])
    same = ts.predict(state, "Team 0", "Team 3", "2026-03-01")
    nxt = ts.predict(state, "Team 0", "Team 3", "2026-10-25")
    assert nxt["rating_home"] == pytest.approx(same["rating_home"] * ts.PARAMS["carry"])


def test_fit_probability_recovers_a_known_logistic():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 8, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.1 + 0.13 * x)))).astype(float)
    a, b = ts.fit_probability(x, y)
    assert a == pytest.approx(0.1, abs=0.05) and b == pytest.approx(0.13, abs=0.01)


def test_state_round_trip(tmp_path):
    state = ts.build_state(_logs(), fit_seasons=["2025-26"])
    path = str(tmp_path / "ts.json")
    ts.save_state(state, path)
    assert ts.load_state(path)["coef"] == state["coef"]


def test_find_team_by_nickname():
    state = {"teams": {"1": {"name": "LA Clippers"}, "2": {"name": "Los Angeles Lakers"}}}
    assert ts.find_team(state, "Los Angeles Clippers")["name"] == "LA Clippers"
    assert ts.find_team(state, "Lakers")["name"] == "Los Angeles Lakers"
