"""
Market maths and the market-odds lookup.

Maths (pure functions, tested in tests/test_market.py): American ↔ decimal
odds, proportional de-vig, fair moneyline from a probability, edge, Kelly
stake at a real price, and cover/over probabilities from MC samples.
De-vig and fair-odds formulas follow linedrift's src/lib/odds-math.ts (MIT,
github.com/souliN02/linedrift), written here from the formulas.

lookup(): market odds for a matchup. The Odds API (one cached call for all
games, 10 min) when a key is configured, else or on failure the ESPN
scoreboard (whose spread/total are also used next to Odds API prices).
"""
import logging
import time
from datetime import date, datetime

import numpy as np

from data import cache
from data.odds import fetch_odds_api_all
from engine.betting import kelly_criterion
from server import clock, config_store
from server.errors import ApiError
from server.jsonutil import r1, r4
from server.security import redact
from server.services import espn, teams

log = logging.getLogger("server")

ODDS_API_TTL = 10 * 60

# Data sources (replaced by demo.install()).
_fetch_odds_api_all = fetch_odds_api_all
_cache              = cache


# ── maths ────────────────────────────────────────────────────────────────────

def american_to_decimal(ml: float) -> float:
    """+150 → 2.5, −200 → 1.5."""
    ml = float(ml)
    if -100 < ml < 100:
        raise ValueError(f"not an American price: {ml}")
    return 1 + ml / 100 if ml > 0 else 1 + 100 / -ml


def decimal_to_american(dec: float) -> float:
    """2.5 → +150, 1.5 → −200."""
    dec = float(dec)
    if dec <= 1:
        raise ValueError(f"decimal odds must be above 1: {dec}")
    return (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)


def implied_prob(ml: float) -> float:
    """Bookmaker's implied probability of an American price (includes the margin)."""
    return 1 / american_to_decimal(ml)


def devig(home_ml: float, away_ml: float) -> dict:
    """Proportional de-vig: fair_i = implied_i / Σ implied."""
    ih, ia = implied_prob(home_ml), implied_prob(away_ml)
    total = ih + ia
    return {
        "implied": {"home": ih, "away": ia},
        "overround": total - 1,
        "fair": {"home": ih / total, "away": ia / total},
    }


def fair_moneyline(p: float) -> float | None:
    """American line matching probability p with no margin: −100p/(1−p) if p ≥ 0.5, else 100(1−p)/p."""
    if p is None or not 0 < p < 1:
        return None
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def edge(model_p: float, fair_p: float) -> float:
    """Model probability minus the market's fair probability."""
    return model_p - fair_p


def cover_prob(margins: np.ndarray, home_spread: float) -> float:
    """P(home covers) = P(margin > −spread); margin is home minus away."""
    return float(np.mean(np.asarray(margins) > -home_spread))


def over_prob(totals: np.ndarray, line: float) -> float:
    """P(total > line)."""
    return float(np.mean(np.asarray(totals) > line))


def kelly_stake(p: float, decimal_odds: float, fraction: float, cap_pct: float) -> dict:
    """Full and fractional Kelly (% of bankroll) at a real price; the fraction is capped."""
    full, frac = kelly_criterion(p, decimal_odds, fraction)
    return {"decimal": round(decimal_odds, 4), "full_pct": full,
            "frac_pct": min(frac, cap_pct), "capped": frac > cap_pct}


# ── market lookup ────────────────────────────────────────────────────────────

def _odds_api_games() -> tuple | None:
    """(games, stored_at) from The Odds API (cached 10 min), or None without a key or on failure."""
    key = config_store.secret("odds_api_key")
    if not key:
        return None
    hit = _cache.get_with_age("web:odds_api_all")
    if hit is not None:
        return hit
    try:
        games = _fetch_odds_api_all(key)
    except Exception as e:
        log.warning("The Odds API failed, falling back to ESPN: %s", redact(str(e)))
        return None
    _cache.set("web:odds_api_all", games, ODDS_API_TTL)
    return games, time.time()


def _odds_api_match(games: list, home: dict, away: dict, d: date) -> dict | None:
    """Best h2h line for the matchup on an ET date from Odds API games."""
    for g in games or []:
        h, a = teams.resolve(g.get("home_team", "")), teams.resolve(g.get("away_team", ""))
        if not h or not a or h["id"] != home["id"] or a["id"] != away["id"]:
            continue
        start = g.get("commence_time")
        if start:
            try:
                tip = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(clock.ET)
                if tip.date() != d:
                    continue
            except ValueError:
                pass
        for bm in g.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                prices = {o.get("name"): o.get("price") for o in mkt.get("outcomes", [])}
                hml, aml = prices.get(g.get("home_team")), prices.get(g.get("away_team"))
                if isinstance(hml, (int, float)) and isinstance(aml, (int, float)):
                    return {"home_ml": int(hml), "away_ml": int(aml),
                            "book": bm.get("title") or "The Odds API"}
    return None


def lookup(home: dict, away: dict, d: date) -> dict:
    """
    Market odds for a matchup on an ET date:
        {status: "ok"|"no_odds_yet"|"not_found", source, fetched_at, stale,
         home_ml, away_ml, spread (home side), total, event}
    `event` is the ESPN scoreboard event (or None); callers drop it from
    public responses. Raises upstream_unavailable only when ESPN is down with
    nothing cached and no Odds API price was found.
    """
    out = {"status": "not_found", "source": None, "fetched_at": None, "stale": False,
           "home_ml": None, "away_ml": None, "spread": None, "total": None, "event": None}

    board, board_err = None, None
    try:
        board = espn.scoreboard(d)
    except ApiError as e:
        board_err = e
    event = espn.find_event(board, home["id"], away["id"]) if board else None
    espn_odds = (event or {}).get("odds") or {}
    out["event"] = event
    out["spread"], out["total"] = espn_odds.get("spread"), espn_odds.get("total")

    api = _odds_api_games()
    if api is not None:
        games, stored_at = api
        line = _odds_api_match(games, home, away, d)
        if line:
            out.update(status="ok", source=f"{line['book']} via The Odds API",
                       fetched_at=espn.iso_ts(stored_at),
                       home_ml=line["home_ml"], away_ml=line["away_ml"])
            return out

    if board is None:
        raise board_err
    out["fetched_at"], out["stale"] = board["as_of"], board["stale"]
    if event is None:
        return out
    if espn_odds.get("home_ml") is not None and espn_odds.get("away_ml") is not None:
        out.update(status="ok", source=f"{espn_odds['provider']} via ESPN",
                   home_ml=espn_odds["home_ml"], away_ml=espn_odds["away_ml"])
    else:
        out["status"] = "no_odds_yet"
        out["source"] = "ESPN"
    return out


def public_odds(found: dict) -> dict:
    """The /api/odds response from a lookup() result."""
    return {k: found[k] for k in ("status", "source", "fetched_at", "stale",
                                  "home_ml", "away_ml", "spread", "total")}


# ── blocks for /api/predict ──────────────────────────────────────────────────

def market_block(found: dict, p_home: float, margins, totals, value_edge: float) -> dict:
    """The predict response's `market` object for a lookup() result or manual prices."""
    hml, aml = found.get("home_ml"), found.get("away_ml")
    spread, total = found.get("spread"), found.get("total")
    block = {
        "status": found["status"], "source": found.get("source"),
        "fetched_at": found.get("fetched_at"), "stale": bool(found.get("stale")),
        "home_ml": hml, "away_ml": aml,
        "decimal": None, "implied": None, "overround": None, "fair": None,
        "spread": r1(spread), "total": r1(total),
        "edge": None, "value_side": None,
        "cover_prob": r4(cover_prob(margins, spread)) if spread is not None else None,
        "over_prob": r4(over_prob(totals, total)) if total is not None else None,
    }
    if hml is not None and aml is not None:
        dv = devig(hml, aml)
        edges = {"home": edge(p_home, dv["fair"]["home"]), "away": edge(1 - p_home, dv["fair"]["away"])}
        best = max(edges, key=edges.get)
        block.update(
            decimal={"home": r4(american_to_decimal(hml)), "away": r4(american_to_decimal(aml))},
            implied={k: r4(v) for k, v in dv["implied"].items()},
            overround=r4(dv["overround"]),
            fair={k: r4(v) for k, v in dv["fair"].items()},
            edge={k: r4(v) for k, v in edges.items()},
            value_side=best if edges[best] >= value_edge else None,
        )
    return block


def kelly_block(p_home: float, market: dict | None, kelly: dict) -> dict | None:
    """Kelly stakes for both sides at the market or manual price; None without a price."""
    if not market or market.get("home_ml") is None or market.get("away_ml") is None:
        return None
    fraction, cap = float(kelly["fraction"]), float(kelly["cap_pct"])
    return {
        "fraction": fraction, "cap_pct": cap,
        "price_source": "manual" if market["status"] == "manual" else "market",
        "home": kelly_stake(p_home, american_to_decimal(market["home_ml"]), fraction, cap),
        "away": kelly_stake(1 - p_home, american_to_decimal(market["away_ml"]), fraction, cap),
    }


def fair_line(p_home: float) -> dict:
    return {"home_ml": r1(fair_moneyline(p_home)), "away_ml": r1(fair_moneyline(1 - p_home))}
