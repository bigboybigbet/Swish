import pandas as pd
import requests


def get_schedule_by_date(d) -> pd.DataFrame | None:
    ds = d.strftime("%Y%m%d")
    try:
        data = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ds}",
            timeout=10,
        ).json()
        rows = []
        for ev in data.get("events", []):
            try:
                c = ev["competitions"][0]["competitors"]
                rows.append({
                    "Date":      d.strftime("%Y-%m-%d"),
                    "Home Team": next(x["team"]["displayName"] for x in c if x["homeAway"] == "home"),
                    "Away Team": next(x["team"]["displayName"] for x in c if x["homeAway"] == "away"),
                })
            except Exception:
                continue
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def get_past_results_espn(d) -> list:
    ds = d.strftime("%Y%m%d")
    try:
        data = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ds}",
            timeout=10,
        ).json()
        out = []
        for ev in data.get("events", []):
            status = ev.get("status", {})
            if status.get("type", {}).get("completed", False) or \
               "FINAL" in status.get("type", {}).get("name", ""):
                comp = ev["competitions"][0]
                c    = comp["competitors"]
                hc   = next(x for x in c if x["homeAway"] == "home")
                ac   = next(x for x in c if x["homeAway"] == "away")
                hs, as_ = int(hc["score"]), int(ac["score"])
                period  = status.get("period", 4)
                out.append({
                    "Home Team":     hc["team"]["displayName"],
                    "Away Team":     ac["team"]["displayName"],
                    "H_Score":       hs,
                    "A_Score":       as_,
                    "Actual_Winner": 1 if hs > as_ else 0,
                    "OT":            period > 4,
                    "Periods":       period,
                })
        return out
    except Exception:
        return []


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
