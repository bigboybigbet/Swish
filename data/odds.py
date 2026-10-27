from datetime import datetime

import requests


def _parse_ml(val):
    """Safely cast ESPN moneyLine (int, float, or string) to int."""
    try:
        return int(float(str(val)))
    except (TypeError, ValueError):
        return None


def fetch_odds_espn(home_team: str, away_team: str, game_date=None):
    """
    Pull moneyline from ESPN scoreboard (no key needed).

    Returns:
        (home_ml, away_ml, source_name)   — odds found
        (None, None, 'game_found')         — game on schedule but no odds yet
        None                               — game not on today's schedule
    """
    try:
        ds = (game_date or datetime.today()).strftime("%Y%m%d")
        data = {}
        for base in [
            "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        ]:
            try:
                data = requests.get(
                    base, params={"dates": ds, "limit": 100}, timeout=8
                ).json()
                break
            except Exception:
                pass

        for ev in data.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home_c = next((c for c in competitors if c["homeAway"] == "home"), None)
            away_c = next((c for c in competitors if c["homeAway"] == "away"), None)
            if not home_c or not away_c:
                continue
            hn = home_c["team"]["displayName"]
            an = away_c["team"]["displayName"]
            if not (home_team.lower() in hn.lower() or hn.lower() in home_team.lower()):
                continue
            if not (away_team.lower() in an.lower() or an.lower() in away_team.lower()):
                continue
            for o in comp.get("odds", []):
                h = o.get("homeTeamOdds") or {}
                a = o.get("awayTeamOdds") or {}
                hml = _parse_ml(
                    h.get("moneyLine") or h.get("current", {}).get("moneyLine")
                )
                aml = _parse_ml(
                    a.get("moneyLine") or a.get("current", {}).get("moneyLine")
                )
                if hml is not None and aml is not None:
                    src = (o.get("provider") or {}).get("name", "ESPN")
                    return hml, aml, src
            return None, None, "game_found"
        return None
    except Exception:
        return None


def fetch_odds_api(home_team: str, away_team: str, api_key: str):
    """
    Fetch from The Odds API (the-odds-api.com).
    Free tier = 500 req/month.
    Returns (home_ml, away_ml, bookmaker_name) or None.
    """
    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/",
            params={
                "apiKey":      api_key,
                "regions":     "us",
                "markets":     "h2h",
                "oddsFormat":  "american",
            },
            timeout=10,
        ).json()
        if not isinstance(resp, list):
            return None
        for game in resp:
            ht_name = game.get("home_team", "")
            at_name = game.get("away_team", "")
            if not (home_team.lower() in ht_name.lower() or ht_name.lower() in home_team.lower()):
                continue
            if not (away_team.lower() in at_name.lower() or at_name.lower() in away_team.lower()):
                continue
            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt["key"] == "h2h":
                        by_name = {o["name"]: o["price"] for o in mkt["outcomes"]}
                        hml = _parse_ml(by_name.get(ht_name))
                        aml = _parse_ml(by_name.get(at_name))
                        if hml is not None and aml is not None:
                            return hml, aml, bm["title"]
        return None
    except Exception:
        return None
