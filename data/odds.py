from datetime import datetime

import requests

from data.schedule import fetch_scoreboard, parse_scoreboard

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"


def _parse_ml(val):
    """Safely cast ESPN moneyLine (int, float, or string) to int."""
    try:
        return int(float(str(val)))
    except (TypeError, ValueError):
        return None


def _names_match(a: str, b: str) -> bool:
    a, b = (a or "").lower(), (b or "").lower()
    return bool(a and b) and (a in b or b in a)


def fetch_odds_espn(home_team: str, away_team: str, game_date=None):
    """
    Pull moneyline from ESPN scoreboard (no key needed).

    Returns:
        (home_ml, away_ml, source_name)   — odds found
        (None, None, 'game_found')         — game on schedule but no odds yet
        None                               — game not on today's schedule
    """
    try:
        events = parse_scoreboard(fetch_scoreboard(game_date or datetime.today()))
        for ev in events:
            if not (_names_match(home_team, ev["home"]["name"])
                    and _names_match(away_team, ev["away"]["name"])):
                continue
            odds = ev["odds"]
            if odds and odds["home_ml"] is not None and odds["away_ml"] is not None:
                return odds["home_ml"], odds["away_ml"], odds["provider"]
            return None, None, "game_found"
        return None
    except Exception:
        return None


def fetch_odds_api_all(api_key: str) -> list:
    """
    Every listed NBA game with its h2h lines from The Odds API, in ONE
    request (the free tier allows 500 a month).  Raises on error.

    The key travels as a query parameter, so exception text contains it:
    redact before showing or logging it.
    """
    resp = requests.get(
        ODDS_API_URL,
        params={
            "apiKey":      api_key,
            "regions":     "us",
            "markets":     "h2h",
            "oddsFormat":  "american",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("unexpected Odds API response")
    return data


def find_odds_api_game(games: list, home_team: str, away_team: str):
    """(home_ml, away_ml, bookmaker_name) for a matchup in fetch_odds_api_all() output, or None."""
    for game in games or []:
        ht_name = game.get("home_team", "")
        at_name = game.get("away_team", "")
        if not (_names_match(home_team, ht_name) and _names_match(away_team, at_name)):
            continue
        for bm in game.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") == "h2h":
                    by_name = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                    hml = _parse_ml(by_name.get(ht_name))
                    aml = _parse_ml(by_name.get(at_name))
                    if hml is not None and aml is not None:
                        return hml, aml, bm.get("title", "")
    return None


def fetch_odds_api(home_team: str, away_team: str, api_key: str):
    """
    Fetch from The Odds API (the-odds-api.com).
    Free tier = 500 req/month.
    Returns (home_ml, away_ml, bookmaker_name) or None.
    """
    try:
        return find_odds_api_game(fetch_odds_api_all(api_key), home_team, away_team)
    except Exception:
        return None
