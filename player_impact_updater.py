"""
Builds player_impact.json from live NBA season stats.
Run from terminal — takes ~10 seconds, updates once a week is plenty.

Usage:
    python player_impact_updater.py              # current season auto-detected
    python player_impact_updater.py 2024-25      # specific season
"""
import os
import sys
import time
from datetime import datetime

import unicodedata
from nba_api.stats.endpoints import leaguedashplayerstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import current_season, season_string  # noqa: E402
from data.atomic import atomic_write_json  # noqa: E402

OUTPUT_FILE = "player_impact.json"
MIN_MINUTES = 150  # filter out players with too little playing time
# Early in a season few players have MIN_MINUTES yet; below this count the
# previous season fills in everyone the current season doesn't cover.
MIN_PLAYERS = 300


def build_player_impact(season: str) -> dict:
    print(f"  Fetching season totals for {season}...")
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="Totals",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0]

    # Keep only players with meaningful playing time
    df = df[df["MIN"] >= MIN_MINUTES].copy()
    print(f"  {len(df)} players after filtering (<{MIN_MINUTES} min removed)")

    # Team totals — sum each team's production from its players
    team_totals = (
        df.groupby("TEAM_ID")[["PTS", "REB", "AST"]]
        .sum()
        .rename(columns={"PTS": "T_PTS", "REB": "T_REB", "AST": "T_AST"})
    )
    df = df.join(team_totals, on="TEAM_ID")

    # Impact = weighted production share, capped at 0.40
    df["IMPACT"] = (
        0.50 * df["PTS"] / df["T_PTS"].clip(lower=1)
        + 0.25 * df["REB"] / df["T_REB"].clip(lower=1)
        + 0.25 * df["AST"] / df["T_AST"].clip(lower=1)
    ).clip(upper=0.40).round(4)

    # Normalize names to ASCII so they match ESPN's spelling (e.g. Jokić → Jokic)
    def _ascii(name):
        return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()

    result = {_ascii(name): round(val, 4)
              for name, val in df.set_index("PLAYER_NAME")["IMPACT"].items()}
    return result


def save(data: dict, season: str):
    payload = {
        "season": season,
        "updated": datetime.today().strftime("%Y-%m-%d %H:%M"),
        "count": len(data),
        "players": data,
    }
    atomic_write_json(OUTPUT_FILE, payload, indent=2)
    print(f"  Saved {len(data)} players to {OUTPUT_FILE}")


def _preview(data: dict):
    top = sorted(data.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n  Top 10 by impact:")
    for name, val in top:
        print(f"    {val:.3f}  {name}")


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else current_season()
    print(f"\nBuilding player impact for season {season}...")
    t0 = time.time()
    data = build_player_impact(season)
    if len(data) < MIN_PLAYERS:
        prev = season_string(int(season[:4]) - 1)
        print(f"  Only {len(data)} players so far — filling gaps from {prev}")
        data = {**build_player_impact(prev), **data}
        season = f"{season} (+{prev})"
    save(data, season)
    _preview(data)
    print(f"\nDone in {time.time()-t0:.1f}s\n")
