"""Real-time pieces: scoreboard cache lifetimes and when ratings rebuild by themselves."""
from datetime import date, datetime, timedelta

from server.services import espn
from server.services.ratings import ratings_stale

TODAY = date(2026, 11, 5)
NOW = datetime(2026, 11, 5, 15, 0)


def test_scoreboard_ttl_short_while_live():
    live = [{"status": "in", "completed": False}, {"status": "pre", "completed": False}]
    assert espn._scoreboard_ttl(TODAY, live) == espn.SCOREBOARD_TTL_LIVE == 30


def test_scoreboard_ttl_past_dates(monkeypatch):
    monkeypatch.setattr(espn.clock, "today_et", lambda: TODAY)
    final = [{"status": "post", "completed": True}]
    unfinished = [{"status": "post", "completed": True}, {"status": "pre", "completed": False}]
    assert espn._scoreboard_ttl(TODAY - timedelta(days=2), final) == espn.SCOREBOARD_TTL_PAST
    assert espn._scoreboard_ttl(TODAY - timedelta(days=1), unfinished) == espn.SCOREBOARD_TTL_TODAY
    assert espn._scoreboard_ttl(TODAY, final) == espn.SCOREBOARD_TTL_TODAY


def test_ratings_stale_rules():
    fresh = {"last_game_date": "2026-11-04", "built_at": "2026-11-05T09:00:00"}
    behind_old = {"last_game_date": "2026-11-02", "built_at": "2026-11-05T09:00:00"}
    behind_new = {"last_game_date": "2026-11-02", "built_at": "2026-11-05T14:00:00"}
    assert ratings_stale(None, TODAY, NOW)
    assert not ratings_stale(fresh, TODAY, NOW)            # yesterday's games are in
    assert ratings_stale(behind_old, TODAY, NOW)           # missing games, rebuilt 6 h ago
    assert not ratings_stale(behind_new, TODAY, NOW)       # rebuilt 1 h ago: wait (offseason / no games)
