import re

import pandas as pd
import requests

SCOREBOARD_URLS = (
    "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
)


# ── ESPN scoreboard: one call serves schedule, results and market odds ────────

def fetch_scoreboard(d) -> dict:
    """Raw ESPN scoreboard JSON for one date.  Raises if every mirror fails."""
    ds = d.strftime("%Y%m%d")
    last_err = None
    for base in SCOREBOARD_URLS:
        try:
            resp = requests.get(base, params={"dates": ds, "limit": 100}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data
            last_err = ValueError("unexpected ESPN scoreboard format")
        except Exception as e:
            last_err = e
    raise last_err


def _num(val):
    """ESPN numbers arrive as int, float or strings like "-140", "o215.5"."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(val))
    return float(m.group()) if m else None


def _ml(val):
    n = _num(val)
    return int(n) if n is not None else None


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _home_spread(o: dict, home_abbr: str, away_abbr: str):
    """Market spread from the home side (negative = home favoured).

    `details` ("LAL -4.5") names the favourite, so it is unambiguous;
    otherwise fall back to the numeric spread fields, which ESPN gives from
    the home side.
    """
    details = (o.get("details") or "").strip()
    m = re.fullmatch(r"([A-Za-z]{2,4})\s+([-+]?\d+(?:\.\d+)?)", details)
    if m:
        abbr, line = m.group(1).upper(), float(m.group(2))
        if abbr == (home_abbr or "").upper():
            return line
        if abbr == (away_abbr or "").upper():
            return -line
    if details.upper() in ("EVEN", "PK", "PICK", "PICK'EM"):
        return 0.0
    for val in (o.get("spread"), _dig(o, "pointSpread", "home", "close", "line")):
        n = _num(val)
        if n is not None:
            return n
    return None


def _parse_odds(comp: dict, home_abbr: str, away_abbr: str):
    """The first odds entry with both moneylines, else the first with a
    spread or total; None if there are none."""
    found = []
    for o in comp.get("odds") or []:
        if not isinstance(o, dict):
            continue
        h = o.get("homeTeamOdds") or {}
        a = o.get("awayTeamOdds") or {}
        hml = _ml(h.get("moneyLine") or (h.get("current") or {}).get("moneyLine")
                  or _dig(o, "moneyline", "home", "close", "odds"))
        aml = _ml(a.get("moneyLine") or (a.get("current") or {}).get("moneyLine")
                  or _dig(o, "moneyline", "away", "close", "odds"))
        spread = _home_spread(o, home_abbr, away_abbr)
        total = _num(o.get("overUnder")) if o.get("overUnder") is not None \
            else _num(_dig(o, "total", "over", "close", "line"))
        if hml is None or aml is None:
            hml = aml = None
        if hml is None and spread is None and total is None:
            continue
        found.append({
            "home_ml":  hml,
            "away_ml":  aml,
            "spread":   spread,
            "total":    total,
            "provider": (o.get("provider") or {}).get("name") or "ESPN",
        })
    with_ml = [x for x in found if x["home_ml"] is not None]
    return (with_ml or found or [None])[0]


def parse_scoreboard(data: dict) -> list:
    """
    One dict per event of a fetch_scoreboard() result:
        event_id, date (ISO UTC tip-off), status "pre"|"in"|"post", completed,
        period, home/away {name, abbr, score|None},
        odds {home_ml, away_ml, spread (home side), total, provider} | None
    Unparseable events are skipped; odds fields are nullable.
    """
    out = []
    for ev in (data or {}).get("events", []):
        try:
            comp = ev["competitions"][0]
            c = comp["competitors"]
            hc = next(x for x in c if x["homeAway"] == "home")
            ac = next(x for x in c if x["homeAway"] == "away")
            st = ev.get("status") or comp.get("status") or {}
            st_type = st.get("type") or {}
            state = st_type.get("state") or "pre"
            completed = bool(st_type.get("completed")) or "FINAL" in (st_type.get("name") or "")
            if completed:
                state = "post"

            def side(x):
                score = _num(x.get("score")) if state != "pre" else None
                return {
                    "name":  x["team"]["displayName"],
                    "abbr":  x["team"].get("abbreviation"),
                    "score": int(score) if score is not None else None,
                }

            home, away = side(hc), side(ac)
            out.append({
                "event_id":  str(ev.get("id")) if ev.get("id") is not None else None,
                "date":      ev.get("date") or comp.get("date"),
                "status":    state,
                "completed": completed,
                "period":    int(st.get("period") or 0),
                "home":      home,
                "away":      away,
                "odds":      _parse_odds(comp, home["abbr"], away["abbr"]),
            })
        except Exception:
            continue
    return out


def get_schedule_by_date(d) -> pd.DataFrame | None:
    try:
        events = parse_scoreboard(fetch_scoreboard(d))
    except Exception:
        return None
    rows = [{
        "Date":      d.strftime("%Y-%m-%d"),
        "Home Team": ev["home"]["name"],
        "Away Team": ev["away"]["name"],
    } for ev in events]
    return pd.DataFrame(rows) if rows else None


def get_past_results_espn(d) -> list:
    try:
        events = parse_scoreboard(fetch_scoreboard(d))
    except Exception:
        return []
    out = []
    for ev in events:
        hs, as_ = ev["home"]["score"], ev["away"]["score"]
        if not ev["completed"] or hs is None or as_ is None:
            continue
        period = ev["period"] or 4
        out.append({
            "Home Team":     ev["home"]["name"],
            "Away Team":     ev["away"]["name"],
            "H_Score":       hs,
            "A_Score":       as_,
            "Actual_Winner": 1 if hs > as_ else 0,
            "OT":            period > 4,
            "Periods":       period,
        })
    return out


def fail_reason(game: dict, pred: int, conf: float) -> str:
    """Short explanation for a wrong prediction."""
    hs, as_ = game["H_Score"], game["A_Score"]
    margin   = abs(hs - as_)
    home_won = hs > as_
    parts    = []

    if game.get("OT"):
        parts.append("2OT" if game.get("Periods", 5) - 4 >= 2 else "OT")

    if margin <= 3:
        parts.append(f"1-possession game (±{margin})")
    elif margin <= 7:
        parts.append(f"clutch finish (±{margin})")
    elif margin <= 14:
        parts.append(f"competitive (±{margin})")
    else:
        parts.append(f"blowout (±{margin})")

    if pred == 1 and not home_won:
        parts.append("away upset")
    elif pred == 0 and home_won:
        parts.append("home stand")

    if conf >= 0.68:
        parts.append(f"conf {conf:.0%}")

    return " · ".join(parts)
