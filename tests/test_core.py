"""Fast offline tests for season logic, Elo and the matchup feature builder."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config import FEATURE_COLS, current_season, recent_seasons
from data.elo import elo_expected, update_elo_game
from engine.models import build_matchup_dataset


# ── seasons ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("day, season", [
    (datetime(2026, 10, 27), "2026-27"),
    (datetime(2027, 3, 1),   "2026-27"),
    (datetime(2026, 9, 30),  "2025-26"),
    (datetime(1999, 11, 2),  "1999-00"),
])
def test_current_season(day, season):
    assert current_season(day) == season


def test_recent_seasons_end_with_current():
    assert recent_seasons(3, datetime(2026, 10, 27)) == ["2024-25", "2025-26", "2026-27"]


# ── Elo ───────────────────────────────────────────────────────────────────────

def test_elo_is_zero_sum_and_rewards_upsets():
    elo = {"A": 1500.0, "B": 1600.0}
    update_elo_game("A", "B", home_won=True, margin=10, elo_dict=elo)
    assert elo["A"] > 1500 and elo["B"] < 1600
    assert elo["A"] + elo["B"] == pytest.approx(3100.0, abs=0.02)


def test_bigger_win_moves_rating_more():
    small, big = {"A": 1500.0, "B": 1500.0}, {"A": 1500.0, "B": 1500.0}
    update_elo_game("A", "B", True, 2, small)
    update_elo_game("A", "B", True, 25, big)
    assert big["A"] > small["A"]


def test_elo_expected_symmetry():
    assert elo_expected(1600, 1500) + elo_expected(1500, 1600) == pytest.approx(1.0)


# ── feature builder ───────────────────────────────────────────────────────────

def _fake_logs(n_days: int = 40, seed: int = 0) -> pd.DataFrame:
    """Four teams, one game per team every other day, team 0 the strongest."""
    rng = np.random.default_rng(seed)
    strength = {0: 8.0, 1: 3.0, 2: -3.0, 3: -8.0}
    rows, gid = [], 1
    for d in range(0, n_days, 2):
        date = pd.Timestamp("2025-10-21") + pd.Timedelta(days=d)
        order = rng.permutation(4)
        for h, a in ((order[0], order[1]), (order[2], order[3])):
            margin = round(strength[h] - strength[a] + 2.5 + rng.normal(0, 10))
            margin = margin or 1
            pts_h = 110 + margin // 2
            pts_a = pts_h - margin
            for team, opp, pts, pm, home in ((h, a, pts_h, margin, True),
                                              (a, h, pts_a, -margin, False)):
                rows.append({
                    "TEAM_ID": 100 + team, "TEAM_NAME": f"Team {team}",
                    "GAME_ID": f"00225{gid:05d}", "GAME_DATE": date,
                    "MATCHUP": f"T{team} vs. T{opp}" if home else f"T{team} @ T{opp}",
                    "WL": "W" if pm > 0 else "L", "PTS": pts, "PLUS_MINUS": float(pm),
                    "FGM": 40, "FGA": 88, "FG3M": 12, "FTA": 22, "AST": 25,
                    "TOV": 13, "REB": 44, "OREB": 10, "DREB": 34,
                })
            gid += 1
    return pd.DataFrame(rows)


def test_build_matchup_dataset_shapes_and_columns():
    df, elo = build_matchup_dataset(_fake_logs(), report=lambda m: None)
    assert len(df) > 0
    assert set(FEATURE_COLS) <= set(df.columns)
    assert {"RESULT", "MARGIN", "GAME_ID"} <= set(df.columns)
    assert df["RESULT"].isin([0, 1]).all()
    assert set(elo) == {f"Team {i}" for i in range(4)}


def test_strongest_team_ends_with_highest_elo():
    _, elo = build_matchup_dataset(_fake_logs(n_days=120), report=lambda m: None)
    assert max(elo, key=elo.get) == "Team 0"
    assert min(elo, key=elo.get) == "Team 3"


def test_features_do_not_see_the_game_itself():
    """Changing a game's score must not change that game's own features."""
    logs = _fake_logs(n_days=60)
    base, _ = build_matchup_dataset(logs, report=lambda m: None)
    gid = base["GAME_ID"].iloc[len(base) // 2]
    tampered = logs.copy()
    mask = tampered["GAME_ID"] == gid
    tampered.loc[mask, "PLUS_MINUS"] *= -1
    tampered.loc[mask, "WL"] = tampered.loc[mask, "WL"].map({"W": "L", "L": "W"})
    after, _ = build_matchup_dataset(tampered, report=lambda m: None)
    row_b = base.set_index("GAME_ID").loc[gid, FEATURE_COLS]
    row_a = after.set_index("GAME_ID").loc[gid, FEATURE_COLS]
    pd.testing.assert_series_equal(row_b, row_a)
