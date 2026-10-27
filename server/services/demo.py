"""
Demo mode: NBA_QUANT_DEMO_DATE=YYYY-MM-DD replays a past date from the
historical CSVs (NBA_HISTORY_DIR, nba_api leaguegamefinder format; loaded
with research.benchmark.load_history).

install() swaps the service layer's data sources; the live code paths are
never changed, and nothing demo-made reaches nba_cache.db:

    ESPN scoreboard   → the date's games from the team logs: finals before
                        the demo date, scheduled on/after it; no odds
    ESPN injury feed  → empty (no injuries)
    The Odds API      → nothing
    nba_api team logs → the historical logs, cut off before the demo date
    Elo ratings       → replayed from the logs of the 3 seasons ending with
                        the demo season, up to the demo date
    caches            → one in-memory cache

server/clock.py already makes today_et() return the demo date.
"""
import logging
import os
import threading
import time
from datetime import date, datetime, time as dtime, timedelta
from functools import lru_cache

import pandas as pd

import config
from data.elo import update_elo_game
from server import clock
from server.services import espn, market, predict, teams

log = logging.getLogger("server")

VALID_PREFIXES = ("002", "004", "005")      # regular season, playoffs, play-in
TIPOFF_ET = dtime(19, 0)                     # the CSVs have dates only

_patches: list = []


def history_dir() -> str:
    from research.benchmark import DEFAULT_DATA_DIR
    return os.environ.get("NBA_HISTORY_DIR") or DEFAULT_DATA_DIR


@lru_cache(maxsize=1)
def history() -> pd.DataFrame:
    """All historical team game logs (one row per team per game)."""
    from research.benchmark import load_history
    raw = load_history(history_dir())
    raw = raw[raw["GAME_ID"].str[:3].isin(VALID_PREFIXES)].copy()
    raw["IS_HOME"] = raw["MATCHUP"].str.contains("vs.", regex=False)
    return raw.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def demo_date() -> date:
    return clock.demo_date()


# ── in-memory cache (same interface as data.cache) ───────────────────────────

class MemoryCache:
    """get / set / get_with_age like data.cache, kept in this process only."""

    def __init__(self):
        self._rows = {}
        self._lock = threading.Lock()

    def get(self, key):
        hit = self.get_with_age(key)
        return hit[0] if hit else None

    def get_with_age(self, key, allow_stale=False):
        with self._lock:
            row = self._rows.get(key)
        if row and (allow_stale or row[1] > time.time()):
            return row[0], row[2]
        return None

    def set(self, key, value, ttl=6 * 3600, stored_at=None):
        now = time.time()
        with self._lock:
            self._rows[key] = (value, now + ttl, stored_at or now)


# ── sources ──────────────────────────────────────────────────────────────────

def _team(row) -> dict | None:
    return teams.get(row["TEAM_ID"]) or teams.resolve(row["TEAM_NAME"])


def _periods(minutes) -> int:
    """Team minutes 48 → 4 periods, 53 → 5 (one OT), …"""
    try:
        return 4 + max(0, round((float(minutes) - 48) / 5))
    except (TypeError, ValueError):
        return 4


def fetch_scoreboard(d) -> dict:
    """ESPN-shaped scoreboard for date d built from the logs (parsed by data.schedule)."""
    d = d if isinstance(d, date) and not isinstance(d, datetime) else d.date()
    raw = history()
    day = raw[raw["GAME_DATE"] == pd.Timestamp(d)]
    final = d < demo_date()
    tip = datetime.combine(d, TIPOFF_ET, clock.ET)
    events = []
    for gid, g in day.groupby("GAME_ID", sort=True):
        home, away = g[g["IS_HOME"]], g[~g["IS_HOME"]]
        if home.empty or away.empty:
            continue
        h, a = home.iloc[0], away.iloc[0]
        ht, at = _team(h), _team(a)
        if not ht or not at:
            continue

        def competitor(row, team, side):
            return {"homeAway": side, "score": str(int(row["PTS"])) if final else None,
                    "team": {"displayName": team["espn_name"], "abbreviation": team["espn_abbr"]}}

        events.append({
            "id": gid,
            "date": clock.iso_utc(tip),
            "name": f"{at['name']} at {ht['name']}",
            "status": {
                "period": _periods(h["MIN"]) if final else 0,
                "type": {"state": "post" if final else "pre", "completed": final,
                         "name": "STATUS_FINAL" if final else "STATUS_SCHEDULED"},
            },
            "competitions": [{"competitors": [competitor(h, ht, "home"), competitor(a, at, "away")],
                              "odds": []}],
        })
    return {"events": events, "demo": True}


def fetch_injury_feed() -> dict:
    """No injury data for replayed dates."""
    return {"injuries": [], "demo": True}


def fetch_odds_api_all(api_key) -> list:
    """Never call The Odds API in demo mode."""
    return []


class _LeagueGameFinder:
    """Stands in for nba_api's LeagueGameFinder: one team's logs before the demo date."""

    def __init__(self, team_id_nullable=None, **_kwargs):
        raw = history()
        mask = raw["GAME_DATE"] < pd.Timestamp(demo_date())
        if team_id_nullable is not None:
            mask &= raw["TEAM_ID"] == int(team_id_nullable)
        cols = [c for c in raw.columns if c != "IS_HOME"]
        self._df = raw.loc[mask, cols].copy()
        self._df["GAME_DATE"] = self._df["GAME_DATE"].dt.strftime("%Y-%m-%d")

    def get_data_frames(self):
        return [self._df.copy()]


class _FinderModule:
    LeagueGameFinder = _LeagueGameFinder


@lru_cache(maxsize=4)
def _elo_as_of(d: date) -> dict:
    """Elo by team id replayed (like build_matchup_dataset) over the 3 seasons ending with d's season."""
    seasons = config.recent_seasons(config.N_TRAIN_SEASONS, datetime.combine(d, dtime()))
    raw = history()
    raw = raw[(raw["GAME_DATE"] < pd.Timestamp(d)) & raw["SEASON_YEAR"].isin(seasons)]
    home = raw[raw["IS_HOME"]][["GAME_ID", "GAME_DATE", "TEAM_ID", "WL", "PLUS_MINUS"]]
    away = raw[~raw["IS_HOME"]][["GAME_ID", "TEAM_ID"]]
    games = home.merge(away, on="GAME_ID", suffixes=("_H", "_A")).sort_values(["GAME_DATE", "GAME_ID"])
    elo: dict = {}
    for r in games.itertuples(index=False):
        update_elo_game(r.TEAM_ID_H, r.TEAM_ID_A, r.WL == "W", float(r.PLUS_MINUS), elo)
    return elo


def elo_ratings() -> dict:
    """The Elo file as of the demo date: {team name: rating}."""
    out = {}
    for team_id, rating in _elo_as_of(demo_date()).items():
        team = teams.get(team_id)
        if team:
            out[team["name"]] = rating
    return out


# ── install / uninstall ──────────────────────────────────────────────────────

def _targets(mem: MemoryCache) -> list:
    from engine import features
    return [
        (espn, "_fetch_scoreboard", fetch_scoreboard),
        (espn, "_fetch_injury_feed", fetch_injury_feed),
        (espn, "_cache", mem),
        (market, "_fetch_odds_api_all", fetch_odds_api_all),
        (market, "_cache", mem),
        (predict, "_load_elo", elo_ratings),
        (predict, "_cache", mem),
        (features, "leaguegamefinder", _FinderModule),
        (features, "cache", mem),
    ]


def active() -> bool:
    return bool(_patches)


def install():
    """Swap every data source for its historical stand-in (idempotent)."""
    if _patches:
        return
    d = demo_date()
    if d is None:
        raise RuntimeError("demo.install() needs NBA_QUANT_DEMO_DATE")
    folder = history_dir()
    if not os.path.isdir(folder):
        raise RuntimeError(f"Demo mode needs the historical game logs in {folder} "
                           f"(set NBA_HISTORY_DIR).")
    for mod, name, new in _targets(MemoryCache()):
        _patches.append((mod, name, getattr(mod, name)))
        setattr(mod, name, new)
    # Load the CSVs in the background so the first request is quick.
    threading.Thread(target=_warm, args=(d,), name="demo-warm", daemon=True).start()


def _warm(d: date):
    try:
        raw = history()
        first, last = raw["GAME_DATE"].min().date(), raw["GAME_DATE"].max().date()
        if not first <= d <= last + timedelta(days=1):
            log.warning("Demo date %s is outside the history (%s to %s): no games to replay",
                        d, first, last)
        _elo_as_of(d)
    except Exception:
        log.exception("Demo mode: could not load the historical data")


def uninstall():
    """Restore the live data sources."""
    while _patches:
        mod, name, old = _patches.pop()
        setattr(mod, name, old)


def info() -> dict | None:
    """{date, history_dir, first, last} for /api/demo, or None in live mode."""
    d = demo_date()
    if d is None:
        return None
    raw = history()
    return {"date": d.isoformat(), "history_dir": history_dir(),
            "first": raw["GAME_DATE"].min().date().isoformat(),
            "last": raw["GAME_DATE"].max().date().isoformat()}
